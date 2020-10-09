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
    
      
      
    
      var element = document.getElementById("fc3e0cc4-f0a4-4610-963d-3ee6c22fae64");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid 'fc3e0cc4-f0a4-4610-963d-3ee6c22fae64' but no matching script tag was found.")
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
                    
                  var docs_json = '{"c9429ac5-3445-4eb1-abfd-d2e3e71cd024":{"roots":{"references":[{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26743","type":"VBar"},{"attributes":{},"id":"26822","type":"UnionRenderers"},{"attributes":{"source":{"id":"26741"}},"id":"26745","type":"CDSView"},{"attributes":{},"id":"26823","type":"Selection"},{"attributes":{"line_dash":[6],"location":0.41666666666666663},"id":"26746","type":"Span"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"26688"},{"id":"26689"},{"id":"26690"},{"id":"26691"},{"id":"26692"},{"id":"26693"},{"id":"26694"},{"id":"26695"}]},"id":"26698","type":"Toolbar"},{"attributes":{"bottom":{"value":1},"fill_alpha":{"value":0.1},"fill_color":{"value":"#ff7f0e"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26749","type":"VBar"},{"attributes":{"data_source":{"id":"26741"},"glyph":{"id":"26742"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26743"},"selection_glyph":null,"view":{"id":"26745"}},"id":"26744","type":"GlyphRenderer"},{"attributes":{"bottom":{"value":2},"fill_color":{"value":"#2ca02c"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26754","type":"VBar"},{"attributes":{"data":{"top":{"__ndarray__":"AAAAAAAA8D8OdNpApw30PxSuR+F6FPY/1AY6baDT9T8c6LSBThv4PxdLfrHkF/c/1QY6baDT9T+V/GLJL5b2P1jyiyW/WPc/43oUrkfh+T8ehetRuB75PxdLfrHkF/c/mJmZmZmZ9z8YrkfhehT2P1RVVVVVVfY/lfxiyS+W9j/gehSuR+H5P5iZmZmZmfc/kl8s+cWS9T8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26806"},"selection_policy":{"id":"26805"}},"id":"26747","type":"ColumnDataSource"},{"attributes":{"bottom":{"value":1},"fill_color":{"value":"#ff7f0e"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26748","type":"VBar"},{"attributes":{"source":{"id":"26747"}},"id":"26751","type":"CDSView"},{"attributes":{"bottom":{"value":2},"fill_alpha":{"value":0.1},"fill_color":{"value":"#2ca02c"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26755","type":"VBar"},{"attributes":{"axis":{"id":"26680"},"ticker":null},"id":"26683","type":"Grid"},{"attributes":{"bottom":{"value":3},"fill_color":{"value":"#d62728"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26760","type":"VBar"},{"attributes":{},"id":"26689","type":"PanTool"},{"attributes":{"axis":{"id":"26714"},"ticker":null},"id":"26717","type":"Grid"},{"attributes":{"data_source":{"id":"26747"},"glyph":{"id":"26748"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26749"},"selection_glyph":null,"view":{"id":"26751"}},"id":"26750","type":"GlyphRenderer"},{"attributes":{},"id":"26710","type":"LinearScale"},{"attributes":{},"id":"26824","type":"UnionRenderers"},{"attributes":{},"id":"26674","type":"DataRange1d"},{"attributes":{},"id":"26825","type":"Selection"},{"attributes":{"callback":null},"id":"26729","type":"HoverTool"},{"attributes":{"line_dash":[6],"location":1.4166666666666665},"id":"26752","type":"Span"},{"attributes":{"axis_label":"Chain","formatter":{"id":"26813"},"ticker":{"id":"26793"}},"id":"26718","type":"LinearAxis"},{"attributes":{"axis_label":"Rank (all chains)","formatter":{"id":"26815"},"major_label_overrides":{"0":"0","1":"1","2":"2","3":"3"},"ticker":{"id":"26715"}},"id":"26714","type":"LinearAxis"},{"attributes":{"data":{"top":{"__ndarray__":"AAAAAAAAAECkcD0K1yMBQCa/WPKLpQFA6LSBThtoAkBqA5020OkCQCz5xZJfrANA8O7u7u5uBEDrUbgehWsDQOtRuB6FawNAC9ejcD0KA0BKfrHkF0sDQA102kCnDQRAqqqqqqoqA0BQG+i0gU4EQC+W/GLJrwRAThvotIFOBECuR+F6FC4EQE4b6LSBTgRAcD0K16PwBEA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26808"},"selection_policy":{"id":"26807"}},"id":"26753","type":"ColumnDataSource"},{"attributes":{},"id":"26715","type":"BasicTicker"},{"attributes":{"source":{"id":"26753"}},"id":"26757","type":"CDSView"},{"attributes":{"bottom":{"value":3},"fill_alpha":{"value":0.1},"fill_color":{"value":"#d62728"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26761","type":"VBar"},{"attributes":{},"id":"26813","type":"BasicTickFormatter"},{"attributes":{},"id":"26678","type":"LinearScale"},{"attributes":{"line_dash":[6],"location":0.48076923076923067},"id":"26774","type":"Span"},{"attributes":{"axis_label":"Chain","formatter":{"id":"26798"},"ticker":{"id":"26765"}},"id":"26684","type":"LinearAxis"},{"attributes":{"data_source":{"id":"26753"},"glyph":{"id":"26754"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26755"},"selection_glyph":null,"view":{"id":"26757"}},"id":"26756","type":"GlyphRenderer"},{"attributes":{"axis":{"id":"26718"},"dimension":1,"ticker":null},"id":"26721","type":"Grid"},{"attributes":{},"id":"26815","type":"BasicTickFormatter"},{"attributes":{"line_dash":[6],"location":2.4166666666666665},"id":"26758","type":"Span"},{"attributes":{},"id":"26803","type":"UnionRenderers"},{"attributes":{"data":{"top":{"__ndarray__":"ZWZmZmZm7j9OG+i0gU7XP2cDnTbQad8/WfKLJb9Y2j9Bpw102kDTP17JL5b8Yt0/PW2g0wY60T9U8oslv1jaP1ws+cWSX9w/WlVVVVVV2T9SVVVVVVXZPzTQaQOdNuA/ZgOdNtBp3z9m8oslv1jaP0h+seQXS9Y/SH6x5BdL1j84baDTBjrRPz+nDXTaQNM/SH6x5BdL1j8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26804"},"selection_policy":{"id":"26803"}},"id":"26741","type":"ColumnDataSource"},{"attributes":{"data":{"top":{"__ndarray__":"MzMzMzOzDUC4HoXrUTgPQDCW/GLJrwxA0GkDnTbQDEBtoNMGOu0LQOi0gU4baApAzszMzMzMC0DrUbgehWsLQClcj8L1qApAqA102kAnCkBH4XoUrkcKQMaSXyz5xQlA6LSBThtoCkAqXI/C9agKQClcj8L1qApACtejcD0KC0AGOm2g0wYKQMkvlvxiyQpAaQOdNtDpCkA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26810"},"selection_policy":{"id":"26809"}},"id":"26759","type":"ColumnDataSource"},{"attributes":{},"id":"26804","type":"Selection"},{"attributes":{"overlay":{"id":"26730"}},"id":"26724","type":"BoxZoomTool"},{"attributes":{"source":{"id":"26759"}},"id":"26763","type":"CDSView"},{"attributes":{},"id":"26712","type":"LinearScale"},{"attributes":{},"id":"26723","type":"PanTool"},{"attributes":{"axis":{"id":"26684"},"dimension":1,"ticker":null},"id":"26687","type":"Grid"},{"attributes":{},"id":"26722","type":"ResetTool"},{"attributes":{},"id":"26676","type":"LinearScale"},{"attributes":{},"id":"26728","type":"SaveTool"},{"attributes":{"axis_label":"Rank (all chains)","formatter":{"id":"26800"},"major_label_overrides":{"0":"0","1":"1","2":"2","3":"3"},"ticker":{"id":"26681"}},"id":"26680","type":"LinearAxis"},{"attributes":{"data_source":{"id":"26759"},"glyph":{"id":"26760"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26761"},"selection_glyph":null,"view":{"id":"26763"}},"id":"26762","type":"GlyphRenderer"},{"attributes":{},"id":"26725","type":"WheelZoomTool"},{"attributes":{"overlay":{"id":"26731"}},"id":"26726","type":"LassoSelectTool"},{"attributes":{"line_dash":[6],"location":3.4166666666666665},"id":"26764","type":"Span"},{"attributes":{},"id":"26727","type":"UndoTool"},{"attributes":{"source":{"id":"26769"}},"id":"26773","type":"CDSView"},{"attributes":{},"id":"26691","type":"WheelZoomTool"},{"attributes":{"line_dash":[6],"location":3.480769230769231},"id":"26792","type":"Span"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26771","type":"VBar"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26770","type":"VBar"},{"attributes":{},"id":"26805","type":"UnionRenderers"},{"attributes":{"callback":null},"id":"26695","type":"HoverTool"},{"attributes":{"data":{"top":{"__ndarray__":"6YVe6IVe4D9nZmZmZmbeP2dmZmZmZu4/WWqlVmql7D/eyI3cyI3YP7vQC73QC9U/uBM7sRM73T+vEzuxEzvdPyZ2Yid2Ytc/lxu5kRu52T8ZuZEbuZHfP5AbuZEbudk/QS/0Qi/04D8LwQ/8wA/cP5AbuZEbudk/q9RKrdRK4z9BL/RCL/TgPyZ2Yid2Ytc/USu1Uiu1wj8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26819"},"selection_policy":{"id":"26818"}},"id":"26769","type":"ColumnDataSource"},{"attributes":{},"id":"26806","type":"Selection"},{"attributes":{"ticks":[0,1,2,3]},"id":"26793","type":"FixedTicker"},{"attributes":{"bottom":{"value":1},"fill_alpha":{"value":0.1},"fill_color":{"value":"#ff7f0e"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26777","type":"VBar"},{"attributes":{"data_source":{"id":"26769"},"glyph":{"id":"26770"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26771"},"selection_glyph":null,"view":{"id":"26773"}},"id":"26772","type":"GlyphRenderer"},{"attributes":{"toolbar":{"id":"26828"},"toolbar_location":"above"},"id":"26829","type":"ToolbarBox"},{"attributes":{"bottom":{"value":2},"fill_color":{"value":"#2ca02c"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26782","type":"VBar"},{"attributes":{"data":{"top":{"__ndarray__":"P/ADP/AD9z+SG7mRG7n2Py/0Qi/0QvU/eqEXeqEX9D9IbuRGbuT3P4If+IEf+PQ/MPRCL/RC9T+ZmZmZmZn3Pyd2Yid2YvQ/9kIv9EIv+D+4kRu5kRv7P7ATO7ETO/o/oBd6oRd6+D+mF3qhF3r4P1ZqpVZqpfk/9EIv9EIv+D9GbuRGbuT3P07sxE7sxPg/wA/8wA/8+z8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26821"},"selection_policy":{"id":"26820"}},"id":"26775","type":"ColumnDataSource"},{"attributes":{"bottom":{"value":1},"fill_color":{"value":"#ff7f0e"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26776","type":"VBar"},{"attributes":{"toolbars":[{"id":"26698"},{"id":"26732"}],"tools":[{"id":"26688"},{"id":"26689"},{"id":"26690"},{"id":"26691"},{"id":"26692"},{"id":"26693"},{"id":"26694"},{"id":"26695"},{"id":"26722"},{"id":"26723"},{"id":"26724"},{"id":"26725"},{"id":"26726"},{"id":"26727"},{"id":"26728"},{"id":"26729"}]},"id":"26828","type":"ProxyToolbar"},{"attributes":{},"id":"26688","type":"ResetTool"},{"attributes":{"source":{"id":"26775"}},"id":"26779","type":"CDSView"},{"attributes":{"text":"mu"},"id":"26795","type":"Title"},{"attributes":{"bottom":{"value":2},"fill_alpha":{"value":0.1},"fill_color":{"value":"#2ca02c"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26783","type":"VBar"},{"attributes":{"bottom":{"value":3},"fill_color":{"value":"#d62728"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26788","type":"VBar"},{"attributes":{},"id":"26681","type":"BasicTicker"},{"attributes":{"data_source":{"id":"26775"},"glyph":{"id":"26776"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26777"},"selection_glyph":null,"view":{"id":"26779"}},"id":"26778","type":"GlyphRenderer"},{"attributes":{"children":[{"id":"26829"},{"id":"26827"}]},"id":"26830","type":"Column"},{"attributes":{"line_dash":[6],"location":1.4807692307692308},"id":"26780","type":"Span"},{"attributes":{},"id":"26807","type":"UnionRenderers"},{"attributes":{"data":{"top":{"__ndarray__":"4Qd+4Af+BUDVSq3USq0EQBh6oRd6oQJAGHqhF3qhAkAUO7ETOzECQCu1Uiu10gRAd2IndmKnA0DFTuzETuwCQHIjN3IjNwNAJDdyIzfyA0Bu5EZu5MYCQB/4gR/4gQNAxU7sxE7sAkDTC73QCz0EQNALvdALPQRA0Au90As9BEB6oRd6oRcEQIIf+IEf+ARAhl7ohV5oBUA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26823"},"selection_policy":{"id":"26822"}},"id":"26781","type":"ColumnDataSource"},{"attributes":{},"id":"26808","type":"Selection"},{"attributes":{"source":{"id":"26781"}},"id":"26785","type":"CDSView"},{"attributes":{"bottom":{"value":3},"fill_alpha":{"value":0.1},"fill_color":{"value":"#d62728"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26789","type":"VBar"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"26730","type":"BoxAnnotation"},{"attributes":{"data_source":{"id":"26781"},"glyph":{"id":"26782"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26783"},"selection_glyph":null,"view":{"id":"26785"}},"id":"26784","type":"GlyphRenderer"},{"attributes":{"line_dash":[6],"location":2.480769230769231},"id":"26786","type":"Span"},{"attributes":{"below":[{"id":"26680"}],"center":[{"id":"26683"},{"id":"26687"},{"id":"26746"},{"id":"26752"},{"id":"26758"},{"id":"26764"}],"left":[{"id":"26684"}],"output_backend":"webgl","plot_height":331,"plot_width":441,"renderers":[{"id":"26744"},{"id":"26750"},{"id":"26756"},{"id":"26762"}],"title":{"id":"26767"},"toolbar":{"id":"26698"},"toolbar_location":null,"x_range":{"id":"26672"},"x_scale":{"id":"26676"},"y_range":{"id":"26674"},"y_scale":{"id":"26678"}},"id":"26671","subtype":"Figure","type":"Plot"},{"attributes":{"data":{"top":{"__ndarray__":"EPzAD/zACUAg+IEf+IELQMEP/MAPfApAdmIndmKnC0A4ciM3ciMOQIZe6IVeaA1Ah17ohV5oDUDYiZ3YiR0NQD7wAz/wAw9Ae6EXeqEXDEAbuZEbuRELQHZiJ3ZipwtAeqEXeqEXDEB0IzdyIzcLQBu5kRu5EQtAFDuxEzsxCkByIzdyIzcLQBu5kRu5EQtAxU7sxE7sCkA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26825"},"selection_policy":{"id":"26824"}},"id":"26787","type":"ColumnDataSource"},{"attributes":{},"id":"26693","type":"UndoTool"},{"attributes":{"source":{"id":"26787"}},"id":"26791","type":"CDSView"},{"attributes":{"text":"tau"},"id":"26767","type":"Title"},{"attributes":{"overlay":{"id":"26697"}},"id":"26692","type":"LassoSelectTool"},{"attributes":{},"id":"26818","type":"UnionRenderers"},{"attributes":{},"id":"26819","type":"Selection"},{"attributes":{"data_source":{"id":"26787"},"glyph":{"id":"26788"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26789"},"selection_glyph":null,"view":{"id":"26791"}},"id":"26790","type":"GlyphRenderer"},{"attributes":{},"id":"26798","type":"BasicTickFormatter"},{"attributes":{},"id":"26809","type":"UnionRenderers"},{"attributes":{"overlay":{"id":"26696"}},"id":"26690","type":"BoxZoomTool"},{"attributes":{},"id":"26810","type":"Selection"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"26731","type":"PolyAnnotation"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"26696","type":"BoxAnnotation"},{"attributes":{},"id":"26694","type":"SaveTool"},{"attributes":{},"id":"26672","type":"DataRange1d"},{"attributes":{},"id":"26800","type":"BasicTickFormatter"},{"attributes":{"ticks":[0,1,2,3]},"id":"26765","type":"FixedTicker"},{"attributes":{},"id":"26820","type":"UnionRenderers"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26742","type":"VBar"},{"attributes":{},"id":"26821","type":"Selection"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"26697","type":"PolyAnnotation"},{"attributes":{"children":[[{"id":"26671"},0,0],[{"id":"26707"},0,1]]},"id":"26827","type":"GridBox"},{"attributes":{"below":[{"id":"26714"}],"center":[{"id":"26717"},{"id":"26721"},{"id":"26774"},{"id":"26780"},{"id":"26786"},{"id":"26792"}],"left":[{"id":"26718"}],"output_backend":"webgl","plot_height":331,"plot_width":441,"renderers":[{"id":"26772"},{"id":"26778"},{"id":"26784"},{"id":"26790"}],"title":{"id":"26795"},"toolbar":{"id":"26732"},"toolbar_location":null,"x_range":{"id":"26672"},"x_scale":{"id":"26710"},"y_range":{"id":"26674"},"y_scale":{"id":"26712"}},"id":"26707","subtype":"Figure","type":"Plot"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"26722"},{"id":"26723"},{"id":"26724"},{"id":"26725"},{"id":"26726"},{"id":"26727"},{"id":"26728"},{"id":"26729"}]},"id":"26732","type":"Toolbar"}],"root_ids":["26830"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"c9429ac5-3445-4eb1-abfd-d2e3e71cd024","root_ids":["26830"],"roots":{"26830":"fc3e0cc4-f0a4-4610-963d-3ee6c22fae64"}}];
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