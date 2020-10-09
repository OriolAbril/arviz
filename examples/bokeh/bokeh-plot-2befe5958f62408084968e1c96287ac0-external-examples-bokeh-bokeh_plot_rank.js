(function() {
  var fn = function() {
    
    (function(root) {
      function now() {
        return new Date();
      }
    
      var force = false;
    
      if (typeof root._bokeh_onload_callbacks === "undefined" || force === true) {
        root._bokeh_onload_callbacks = [];
        root._bokeh_is_loading = undefined;
      }
    
      
      
    
      var element = document.getElementById("b4fb9733-ae7a-42e6-8f99-923d1e6e2862");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid 'b4fb9733-ae7a-42e6-8f99-923d1e6e2862' but no matching script tag was found.")
        }
      
    
      function run_callbacks() {
        try {
          root._bokeh_onload_callbacks.forEach(function(callback) {
            if (callback != null)
              callback();
          });
        } finally {
          delete root._bokeh_onload_callbacks
        }
        console.debug("Bokeh: all callbacks have finished");
      }
    
      function load_libs(css_urls, js_urls, callback) {
        if (css_urls == null) css_urls = [];
        if (js_urls == null) js_urls = [];
    
        root._bokeh_onload_callbacks.push(callback);
        if (root._bokeh_is_loading > 0) {
          console.debug("Bokeh: BokehJS is being loaded, scheduling callback at", now());
          return null;
        }
        if (js_urls == null || js_urls.length === 0) {
          run_callbacks();
          return null;
        }
        console.debug("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
        root._bokeh_is_loading = css_urls.length + js_urls.length;
    
        function on_load() {
          root._bokeh_is_loading--;
          if (root._bokeh_is_loading === 0) {
            console.debug("Bokeh: all BokehJS libraries/stylesheets loaded");
            run_callbacks()
          }
        }
    
        function on_error() {
          console.error("failed to load " + url);
        }
    
        for (var i = 0; i < css_urls.length; i++) {
          var url = css_urls[i];
          const element = document.createElement("link");
          element.onload = on_load;
          element.onerror = on_error;
          element.rel = "stylesheet";
          element.type = "text/css";
          element.href = url;
          console.debug("Bokeh: injecting link tag for BokehJS stylesheet: ", url);
          document.body.appendChild(element);
        }
    
        const hashes = {"https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js": "qkRvDQVAIfzsJo40iRBbxt6sttt0hv4lh74DG7OK4MCHv4C5oohXYoHUM5W11uqS", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js": "Sb7Mr06a9TNlet/GEBeKaf5xH3eb6AlCzwjtU82wNPyDrnfoiVl26qnvlKjmcAd+", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js": "HaJ15vgfmcfRtB4c4YBOI4f1MUujukqInOWVqZJZZGK7Q+ivud0OKGSTn/Vm2iso"};
    
        for (var i = 0; i < js_urls.length; i++) {
          var url = js_urls[i];
          var element = document.createElement('script');
          element.onload = on_load;
          element.onerror = on_error;
          element.async = false;
          element.src = url;
          if (url in hashes) {
            element.crossOrigin = "anonymous";
            element.integrity = "sha384-" + hashes[url];
          }
          console.debug("Bokeh: injecting script tag for BokehJS library: ", url);
          document.head.appendChild(element);
        }
      };
    
      function inject_raw_css(css) {
        const element = document.createElement("style");
        element.appendChild(document.createTextNode(css));
        document.body.appendChild(element);
      }
    
      
      var js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js"];
      var css_urls = [];
      
    
      var inline_js = [
        function(Bokeh) {
          Bokeh.set_log_level("info");
        },
        
        function(Bokeh) {
          (function() {
            var fn = function() {
              Bokeh.safely(function() {
                (function(root) {
                  function embed_document(root) {
                    
                  var docs_json = '{"c63db31e-c7d8-4ca9-a5bd-56547a613648":{"roots":{"references":[{"attributes":{"text":"mu"},"id":"26842","type":"Title"},{"attributes":{"bottom":{"value":1},"fill_alpha":{"value":0.1},"fill_color":{"value":"#ff7f0e"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26824","type":"VBar"},{"attributes":{"bottom":{"value":2},"fill_color":{"value":"#2ca02c"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26829","type":"VBar"},{"attributes":{"below":[{"id":"26761"}],"center":[{"id":"26764"},{"id":"26768"},{"id":"26821"},{"id":"26827"},{"id":"26833"},{"id":"26839"}],"left":[{"id":"26765"}],"output_backend":"webgl","plot_height":331,"plot_width":441,"renderers":[{"id":"26819"},{"id":"26825"},{"id":"26831"},{"id":"26837"}],"title":{"id":"26842"},"toolbar":{"id":"26779"},"toolbar_location":null,"x_range":{"id":"26719"},"x_scale":{"id":"26757"},"y_range":{"id":"26721"},"y_scale":{"id":"26759"}},"id":"26754","subtype":"Figure","type":"Plot"},{"attributes":{"source":{"id":"26822"}},"id":"26826","type":"CDSView"},{"attributes":{},"id":"26861","type":"BasicTickFormatter"},{"attributes":{"bottom":{"value":2},"fill_alpha":{"value":0.1},"fill_color":{"value":"#2ca02c"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26830","type":"VBar"},{"attributes":{"bottom":{"value":3},"fill_color":{"value":"#d62728"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26835","type":"VBar"},{"attributes":{"data_source":{"id":"26822"},"glyph":{"id":"26823"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26824"},"selection_glyph":null,"view":{"id":"26826"}},"id":"26825","type":"GlyphRenderer"},{"attributes":{"children":[{"id":"26876"},{"id":"26874"}]},"id":"26877","type":"Column"},{"attributes":{"line_dash":[6],"location":1.4807692307692308},"id":"26827","type":"Span"},{"attributes":{"bottom":{"value":1},"fill_color":{"value":"#ff7f0e"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26823","type":"VBar"},{"attributes":{"data":{"top":{"__ndarray__":"4Qd+4Af+BUDVSq3USq0EQBh6oRd6oQJAGHqhF3qhAkAUO7ETOzECQCu1Uiu10gRAd2IndmKnA0DFTuzETuwCQHIjN3IjNwNAJDdyIzfyA0Bu5EZu5MYCQB/4gR/4gQNAxU7sxE7sAkDTC73QCz0EQNALvdALPQRA0Au90As9BEB6oRd6oRcEQIIf+IEf+ARAhl7ohV5oBUA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26870"},"selection_policy":{"id":"26869"}},"id":"26828","type":"ColumnDataSource"},{"attributes":{"toolbars":[{"id":"26745"},{"id":"26779"}],"tools":[{"id":"26735"},{"id":"26736"},{"id":"26737"},{"id":"26738"},{"id":"26739"},{"id":"26740"},{"id":"26741"},{"id":"26742"},{"id":"26769"},{"id":"26770"},{"id":"26771"},{"id":"26772"},{"id":"26773"},{"id":"26774"},{"id":"26775"},{"id":"26776"}]},"id":"26875","type":"ProxyToolbar"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"26777","type":"BoxAnnotation"},{"attributes":{},"id":"26854","type":"UnionRenderers"},{"attributes":{"source":{"id":"26828"}},"id":"26832","type":"CDSView"},{"attributes":{},"id":"26741","type":"SaveTool"},{"attributes":{"bottom":{"value":3},"fill_alpha":{"value":0.1},"fill_color":{"value":"#d62728"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26836","type":"VBar"},{"attributes":{"callback":null},"id":"26742","type":"HoverTool"},{"attributes":{},"id":"26855","type":"Selection"},{"attributes":{"data_source":{"id":"26828"},"glyph":{"id":"26829"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26830"},"selection_glyph":null,"view":{"id":"26832"}},"id":"26831","type":"GlyphRenderer"},{"attributes":{"line_dash":[6],"location":2.480769230769231},"id":"26833","type":"Span"},{"attributes":{"data":{"top":{"__ndarray__":"EPzAD/zACUAg+IEf+IELQMEP/MAPfApAdmIndmKnC0A4ciM3ciMOQIZe6IVeaA1Ah17ohV5oDUDYiZ3YiR0NQD7wAz/wAw9Ae6EXeqEXDEAbuZEbuRELQHZiJ3ZipwtAeqEXeqEXDEB0IzdyIzcLQBu5kRu5EQtAFDuxEzsxCkByIzdyIzcLQBu5kRu5EQtAxU7sxE7sCkA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26872"},"selection_policy":{"id":"26871"}},"id":"26834","type":"ColumnDataSource"},{"attributes":{"source":{"id":"26834"}},"id":"26838","type":"CDSView"},{"attributes":{},"id":"26865","type":"UnionRenderers"},{"attributes":{"data_source":{"id":"26834"},"glyph":{"id":"26835"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26836"},"selection_glyph":null,"view":{"id":"26838"}},"id":"26837","type":"GlyphRenderer"},{"attributes":{"axis":{"id":"26727"},"ticker":null},"id":"26730","type":"Grid"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"26778","type":"PolyAnnotation"},{"attributes":{},"id":"26866","type":"Selection"},{"attributes":{"line_dash":[6],"location":3.480769230769231},"id":"26839","type":"Span"},{"attributes":{},"id":"26856","type":"UnionRenderers"},{"attributes":{"data":{"top":{"__ndarray__":"6YVe6IVe4D9nZmZmZmbeP2dmZmZmZu4/WWqlVmql7D/eyI3cyI3YP7vQC73QC9U/uBM7sRM73T+vEzuxEzvdPyZ2Yid2Ytc/lxu5kRu52T8ZuZEbuZHfP5AbuZEbudk/QS/0Qi/04D8LwQ/8wA/cP5AbuZEbudk/q9RKrdRK4z9BL/RCL/TgPyZ2Yid2Ytc/USu1Uiu1wj8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26866"},"selection_policy":{"id":"26865"}},"id":"26816","type":"ColumnDataSource"},{"attributes":{},"id":"26857","type":"Selection"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"26735"},{"id":"26736"},{"id":"26737"},{"id":"26738"},{"id":"26739"},{"id":"26740"},{"id":"26741"},{"id":"26742"}]},"id":"26745","type":"Toolbar"},{"attributes":{},"id":"26723","type":"LinearScale"},{"attributes":{},"id":"26867","type":"UnionRenderers"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26789","type":"VBar"},{"attributes":{"data_source":{"id":"26816"},"glyph":{"id":"26817"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26818"},"selection_glyph":null,"view":{"id":"26820"}},"id":"26819","type":"GlyphRenderer"},{"attributes":{},"id":"26868","type":"Selection"},{"attributes":{},"id":"26719","type":"DataRange1d"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"26744","type":"PolyAnnotation"},{"attributes":{"axis":{"id":"26731"},"dimension":1,"ticker":null},"id":"26734","type":"Grid"},{"attributes":{},"id":"26845","type":"BasicTickFormatter"},{"attributes":{"axis_label":"Chain","formatter":{"id":"26845"},"ticker":{"id":"26812"}},"id":"26731","type":"LinearAxis"},{"attributes":{},"id":"26728","type":"BasicTicker"},{"attributes":{"axis_label":"Rank (all chains)","formatter":{"id":"26846"},"major_label_overrides":{"0":"0","1":"1","2":"2","3":"3"},"ticker":{"id":"26728"}},"id":"26727","type":"LinearAxis"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"26743","type":"BoxAnnotation"},{"attributes":{"text":"tau"},"id":"26814","type":"Title"},{"attributes":{},"id":"26759","type":"LinearScale"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26790","type":"VBar"},{"attributes":{},"id":"26869","type":"UnionRenderers"},{"attributes":{"line_dash":[6],"location":0.41666666666666663},"id":"26793","type":"Span"},{"attributes":{"source":{"id":"26788"}},"id":"26792","type":"CDSView"},{"attributes":{},"id":"26870","type":"Selection"},{"attributes":{"bottom":{"value":1},"fill_alpha":{"value":0.1},"fill_color":{"value":"#ff7f0e"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26796","type":"VBar"},{"attributes":{"data_source":{"id":"26788"},"glyph":{"id":"26789"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26790"},"selection_glyph":null,"view":{"id":"26792"}},"id":"26791","type":"GlyphRenderer"},{"attributes":{"bottom":{"value":2},"fill_color":{"value":"#2ca02c"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26801","type":"VBar"},{"attributes":{"data":{"top":{"__ndarray__":"AAAAAAAA8D8OdNpApw30PxSuR+F6FPY/1AY6baDT9T8c6LSBThv4PxdLfrHkF/c/1QY6baDT9T+V/GLJL5b2P1jyiyW/WPc/43oUrkfh+T8ehetRuB75PxdLfrHkF/c/mJmZmZmZ9z8YrkfhehT2P1RVVVVVVfY/lfxiyS+W9j/gehSuR+H5P5iZmZmZmfc/kl8s+cWS9T8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26853"},"selection_policy":{"id":"26852"}},"id":"26794","type":"ColumnDataSource"},{"attributes":{"ticks":[0,1,2,3]},"id":"26812","type":"FixedTicker"},{"attributes":{},"id":"26846","type":"BasicTickFormatter"},{"attributes":{"bottom":{"value":1},"fill_color":{"value":"#ff7f0e"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26795","type":"VBar"},{"attributes":{},"id":"26721","type":"DataRange1d"},{"attributes":{"source":{"id":"26794"}},"id":"26798","type":"CDSView"},{"attributes":{"bottom":{"value":2},"fill_alpha":{"value":0.1},"fill_color":{"value":"#2ca02c"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26802","type":"VBar"},{"attributes":{"bottom":{"value":3},"fill_color":{"value":"#d62728"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26807","type":"VBar"},{"attributes":{"axis":{"id":"26761"},"ticker":null},"id":"26764","type":"Grid"},{"attributes":{"line_dash":[6],"location":3.4166666666666665},"id":"26811","type":"Span"},{"attributes":{"callback":null},"id":"26776","type":"HoverTool"},{"attributes":{"data_source":{"id":"26794"},"glyph":{"id":"26795"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26796"},"selection_glyph":null,"view":{"id":"26798"}},"id":"26797","type":"GlyphRenderer"},{"attributes":{},"id":"26871","type":"UnionRenderers"},{"attributes":{"axis_label":"Chain","formatter":{"id":"26860"},"ticker":{"id":"26840"}},"id":"26765","type":"LinearAxis"},{"attributes":{},"id":"26757","type":"LinearScale"},{"attributes":{"axis_label":"Rank (all chains)","formatter":{"id":"26861"},"major_label_overrides":{"0":"0","1":"1","2":"2","3":"3"},"ticker":{"id":"26762"}},"id":"26761","type":"LinearAxis"},{"attributes":{"line_dash":[6],"location":1.4166666666666665},"id":"26799","type":"Span"},{"attributes":{},"id":"26872","type":"Selection"},{"attributes":{},"id":"26762","type":"BasicTicker"},{"attributes":{"data":{"top":{"__ndarray__":"AAAAAAAAAECkcD0K1yMBQCa/WPKLpQFA6LSBThtoAkBqA5020OkCQCz5xZJfrANA8O7u7u5uBEDrUbgehWsDQOtRuB6FawNAC9ejcD0KA0BKfrHkF0sDQA102kCnDQRAqqqqqqoqA0BQG+i0gU4EQC+W/GLJrwRAThvotIFOBECuR+F6FC4EQE4b6LSBTgRAcD0K16PwBEA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26855"},"selection_policy":{"id":"26854"}},"id":"26800","type":"ColumnDataSource"},{"attributes":{"source":{"id":"26800"}},"id":"26804","type":"CDSView"},{"attributes":{"bottom":{"value":3},"fill_alpha":{"value":0.1},"fill_color":{"value":"#d62728"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26808","type":"VBar"},{"attributes":{"overlay":{"id":"26743"}},"id":"26737","type":"BoxZoomTool"},{"attributes":{"line_dash":[6],"location":0.48076923076923067},"id":"26821","type":"Span"},{"attributes":{"below":[{"id":"26727"}],"center":[{"id":"26730"},{"id":"26734"},{"id":"26793"},{"id":"26799"},{"id":"26805"},{"id":"26811"}],"left":[{"id":"26731"}],"output_backend":"webgl","plot_height":331,"plot_width":441,"renderers":[{"id":"26791"},{"id":"26797"},{"id":"26803"},{"id":"26809"}],"title":{"id":"26814"},"toolbar":{"id":"26745"},"toolbar_location":null,"x_range":{"id":"26719"},"x_scale":{"id":"26723"},"y_range":{"id":"26721"},"y_scale":{"id":"26725"}},"id":"26718","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"26736","type":"PanTool"},{"attributes":{"data_source":{"id":"26800"},"glyph":{"id":"26801"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26802"},"selection_glyph":null,"view":{"id":"26804"}},"id":"26803","type":"GlyphRenderer"},{"attributes":{"axis":{"id":"26765"},"dimension":1,"ticker":null},"id":"26768","type":"Grid"},{"attributes":{},"id":"26725","type":"LinearScale"},{"attributes":{"line_dash":[6],"location":2.4166666666666665},"id":"26805","type":"Span"},{"attributes":{"data":{"top":{"__ndarray__":"ZWZmZmZm7j9OG+i0gU7XP2cDnTbQad8/WfKLJb9Y2j9Bpw102kDTP17JL5b8Yt0/PW2g0wY60T9U8oslv1jaP1ws+cWSX9w/WlVVVVVV2T9SVVVVVVXZPzTQaQOdNuA/ZgOdNtBp3z9m8oslv1jaP0h+seQXS9Y/SH6x5BdL1j84baDTBjrRPz+nDXTaQNM/SH6x5BdL1j8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26851"},"selection_policy":{"id":"26850"}},"id":"26788","type":"ColumnDataSource"},{"attributes":{},"id":"26850","type":"UnionRenderers"},{"attributes":{"data":{"top":{"__ndarray__":"MzMzMzOzDUC4HoXrUTgPQDCW/GLJrwxA0GkDnTbQDEBtoNMGOu0LQOi0gU4baApAzszMzMzMC0DrUbgehWsLQClcj8L1qApAqA102kAnCkBH4XoUrkcKQMaSXyz5xQlA6LSBThtoCkAqXI/C9agKQClcj8L1qApACtejcD0KC0AGOm2g0wYKQMkvlvxiyQpAaQOdNtDpCkA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26857"},"selection_policy":{"id":"26856"}},"id":"26806","type":"ColumnDataSource"},{"attributes":{"overlay":{"id":"26777"}},"id":"26771","type":"BoxZoomTool"},{"attributes":{"source":{"id":"26806"}},"id":"26810","type":"CDSView"},{"attributes":{},"id":"26770","type":"PanTool"},{"attributes":{},"id":"26851","type":"Selection"},{"attributes":{},"id":"26769","type":"ResetTool"},{"attributes":{"children":[[{"id":"26718"},0,0],[{"id":"26754"},0,1]]},"id":"26874","type":"GridBox"},{"attributes":{"ticks":[0,1,2,3]},"id":"26840","type":"FixedTicker"},{"attributes":{"toolbar":{"id":"26875"},"toolbar_location":"above"},"id":"26876","type":"ToolbarBox"},{"attributes":{"data_source":{"id":"26806"},"glyph":{"id":"26807"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26808"},"selection_glyph":null,"view":{"id":"26810"}},"id":"26809","type":"GlyphRenderer"},{"attributes":{},"id":"26775","type":"SaveTool"},{"attributes":{},"id":"26772","type":"WheelZoomTool"},{"attributes":{"source":{"id":"26816"}},"id":"26820","type":"CDSView"},{"attributes":{},"id":"26735","type":"ResetTool"},{"attributes":{"overlay":{"id":"26778"}},"id":"26773","type":"LassoSelectTool"},{"attributes":{},"id":"26738","type":"WheelZoomTool"},{"attributes":{},"id":"26774","type":"UndoTool"},{"attributes":{},"id":"26860","type":"BasicTickFormatter"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"26769"},{"id":"26770"},{"id":"26771"},{"id":"26772"},{"id":"26773"},{"id":"26774"},{"id":"26775"},{"id":"26776"}]},"id":"26779","type":"Toolbar"},{"attributes":{"overlay":{"id":"26744"}},"id":"26739","type":"LassoSelectTool"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26818","type":"VBar"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26817","type":"VBar"},{"attributes":{},"id":"26740","type":"UndoTool"},{"attributes":{},"id":"26852","type":"UnionRenderers"},{"attributes":{"data":{"top":{"__ndarray__":"P/ADP/AD9z+SG7mRG7n2Py/0Qi/0QvU/eqEXeqEX9D9IbuRGbuT3P4If+IEf+PQ/MPRCL/RC9T+ZmZmZmZn3Pyd2Yid2YvQ/9kIv9EIv+D+4kRu5kRv7P7ATO7ETO/o/oBd6oRd6+D+mF3qhF3r4P1ZqpVZqpfk/9EIv9EIv+D9GbuRGbuT3P07sxE7sxPg/wA/8wA/8+z8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26868"},"selection_policy":{"id":"26867"}},"id":"26822","type":"ColumnDataSource"},{"attributes":{},"id":"26853","type":"Selection"}],"root_ids":["26877"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"c63db31e-c7d8-4ca9-a5bd-56547a613648","root_ids":["26877"],"roots":{"26877":"b4fb9733-ae7a-42e6-8f99-923d1e6e2862"}}];
                  root.Bokeh.embed.embed_items(docs_json, render_items);
                
                  }
                  if (root.Bokeh !== undefined) {
                    embed_document(root);
                  } else {
                    var attempts = 0;
                    var timer = setInterval(function(root) {
                      if (root.Bokeh !== undefined) {
                        clearInterval(timer);
                        embed_document(root);
                      } else {
                        attempts++;
                        if (attempts > 100) {
                          clearInterval(timer);
                          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                        }
                      }
                    }, 10, root)
                  }
                })(window);
              });
            };
            if (document.readyState != "loading") fn();
            else document.addEventListener("DOMContentLoaded", fn);
          })();
        },
        function(Bokeh) {
        
        
        }
      ];
    
      function run_inline_js() {
        
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i].call(root, root.Bokeh);
        }
        
      }
    
      if (root._bokeh_is_loading === 0) {
        console.debug("Bokeh: BokehJS loaded, going straight to plotting");
        run_inline_js();
      } else {
        load_libs(css_urls, js_urls, function() {
          console.debug("Bokeh: BokehJS plotting callback run at", now());
          run_inline_js();
        });
      }
    }(window));
  };
  if (document.readyState != "loading") fn();
  else document.addEventListener("DOMContentLoaded", fn);
})();