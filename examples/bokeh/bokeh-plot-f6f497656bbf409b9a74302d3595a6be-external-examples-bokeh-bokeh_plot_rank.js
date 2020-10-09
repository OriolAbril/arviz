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
    
      
      
    
      var element = document.getElementById("9bf8fc82-d40e-4fab-8ace-29bb13eee093");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '9bf8fc82-d40e-4fab-8ace-29bb13eee093' but no matching script tag was found.")
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
                    
                  var docs_json = '{"f0b5c336-43c9-41fd-978d-2e1def82afa0":{"roots":{"references":[{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"26625","type":"BoxAnnotation"},{"attributes":{"data":{"top":{"__ndarray__":"6YVe6IVe4D9nZmZmZmbeP2dmZmZmZu4/WWqlVmql7D/eyI3cyI3YP7vQC73QC9U/uBM7sRM73T+vEzuxEzvdPyZ2Yid2Ytc/lxu5kRu52T8ZuZEbuZHfP5AbuZEbudk/QS/0Qi/04D8LwQ/8wA/cP5AbuZEbudk/q9RKrdRK4z9BL/RCL/TgPyZ2Yid2Ytc/USu1Uiu1wj8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26711"},"selection_policy":{"id":"26712"}},"id":"26664","type":"ColumnDataSource"},{"attributes":{"toolbar":{"id":"26723"},"toolbar_location":"above"},"id":"26724","type":"ToolbarBox"},{"attributes":{},"id":"26693","type":"BasicTickFormatter"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26666","type":"VBar"},{"attributes":{},"id":"26622","type":"UndoTool"},{"attributes":{"data_source":{"id":"26654"},"glyph":{"id":"26655"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26656"},"selection_glyph":null,"view":{"id":"26658"}},"id":"26657","type":"GlyphRenderer"},{"attributes":{},"id":"26620","type":"WheelZoomTool"},{"attributes":{"ticks":[0,1,2,3]},"id":"26688","type":"FixedTicker"},{"attributes":{"overlay":{"id":"26626"}},"id":"26621","type":"LassoSelectTool"},{"attributes":{},"id":"26617","type":"ResetTool"},{"attributes":{"source":{"id":"26654"}},"id":"26658","type":"CDSView"},{"attributes":{},"id":"26623","type":"SaveTool"},{"attributes":{"data":{"top":{"__ndarray__":"MzMzMzOzDUC4HoXrUTgPQDCW/GLJrwxA0GkDnTbQDEBtoNMGOu0LQOi0gU4baApAzszMzMzMC0DrUbgehWsLQClcj8L1qApAqA102kAnCkBH4XoUrkcKQMaSXyz5xQlA6LSBThtoCkAqXI/C9agKQClcj8L1qApACtejcD0KC0AGOm2g0wYKQMkvlvxiyQpAaQOdNtDpCkA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26702"},"selection_policy":{"id":"26703"}},"id":"26654","type":"ColumnDataSource"},{"attributes":{"overlay":{"id":"26625"}},"id":"26619","type":"BoxZoomTool"},{"attributes":{"line_dash":[6],"location":2.4166666666666665},"id":"26653","type":"Span"},{"attributes":{},"id":"26618","type":"PanTool"},{"attributes":{"source":{"id":"26636"}},"id":"26640","type":"CDSView"},{"attributes":{"children":[[{"id":"26566"},0,0],[{"id":"26602"},0,1]]},"id":"26722","type":"GridBox"},{"attributes":{"data_source":{"id":"26648"},"glyph":{"id":"26649"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26650"},"selection_glyph":null,"view":{"id":"26652"}},"id":"26651","type":"GlyphRenderer"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"26617"},{"id":"26618"},{"id":"26619"},{"id":"26620"},{"id":"26621"},{"id":"26622"},{"id":"26623"},{"id":"26624"}]},"id":"26627","type":"Toolbar"},{"attributes":{"bottom":{"value":3},"fill_alpha":{"value":0.1},"fill_color":{"value":"#d62728"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26656","type":"VBar"},{"attributes":{"source":{"id":"26664"}},"id":"26668","type":"CDSView"},{"attributes":{},"id":"26697","type":"UnionRenderers"},{"attributes":{"axis_label":"Rank (all chains)","formatter":{"id":"26709"},"major_label_overrides":{"0":"0","1":"1","2":"2","3":"3"},"ticker":{"id":"26610"}},"id":"26609","type":"LinearAxis"},{"attributes":{"data":{"top":{"__ndarray__":"AAAAAAAAAECkcD0K1yMBQCa/WPKLpQFA6LSBThtoAkBqA5020OkCQCz5xZJfrANA8O7u7u5uBEDrUbgehWsDQOtRuB6FawNAC9ejcD0KA0BKfrHkF0sDQA102kCnDQRAqqqqqqoqA0BQG+i0gU4EQC+W/GLJrwRAThvotIFOBECuR+F6FC4EQE4b6LSBTgRAcD0K16PwBEA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26700"},"selection_policy":{"id":"26701"}},"id":"26648","type":"ColumnDataSource"},{"attributes":{"source":{"id":"26648"}},"id":"26652","type":"CDSView"},{"attributes":{"below":[{"id":"26609"}],"center":[{"id":"26612"},{"id":"26616"},{"id":"26669"},{"id":"26675"},{"id":"26681"},{"id":"26687"}],"left":[{"id":"26613"}],"output_backend":"webgl","plot_height":331,"plot_width":441,"renderers":[{"id":"26667"},{"id":"26673"},{"id":"26679"},{"id":"26685"}],"title":{"id":"26690"},"toolbar":{"id":"26627"},"toolbar_location":null,"x_range":{"id":"26567"},"x_scale":{"id":"26605"},"y_range":{"id":"26569"},"y_scale":{"id":"26607"}},"id":"26602","subtype":"Figure","type":"Plot"},{"attributes":{"line_dash":[6],"location":1.4166666666666665},"id":"26647","type":"Span"},{"attributes":{},"id":"26583","type":"ResetTool"},{"attributes":{"bottom":{"value":3},"fill_color":{"value":"#d62728"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26655","type":"VBar"},{"attributes":{"callback":null},"id":"26624","type":"HoverTool"},{"attributes":{"data_source":{"id":"26642"},"glyph":{"id":"26643"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26644"},"selection_glyph":null,"view":{"id":"26646"}},"id":"26645","type":"GlyphRenderer"},{"attributes":{},"id":"26610","type":"BasicTicker"},{"attributes":{"overlay":{"id":"26591"}},"id":"26585","type":"BoxZoomTool"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26665","type":"VBar"},{"attributes":{},"id":"26700","type":"Selection"},{"attributes":{},"id":"26589","type":"SaveTool"},{"attributes":{"bottom":{"value":2},"fill_alpha":{"value":0.1},"fill_color":{"value":"#2ca02c"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26650","type":"VBar"},{"attributes":{"axis":{"id":"26609"},"ticker":null},"id":"26612","type":"Grid"},{"attributes":{"axis_label":"Chain","formatter":{"id":"26693"},"ticker":{"id":"26660"}},"id":"26579","type":"LinearAxis"},{"attributes":{"toolbars":[{"id":"26593"},{"id":"26627"}],"tools":[{"id":"26583"},{"id":"26584"},{"id":"26585"},{"id":"26586"},{"id":"26587"},{"id":"26588"},{"id":"26589"},{"id":"26590"},{"id":"26617"},{"id":"26618"},{"id":"26619"},{"id":"26620"},{"id":"26621"},{"id":"26622"},{"id":"26623"},{"id":"26624"}]},"id":"26723","type":"ProxyToolbar"},{"attributes":{"axis_label":"Chain","formatter":{"id":"26708"},"ticker":{"id":"26688"}},"id":"26613","type":"LinearAxis"},{"attributes":{"line_dash":[6],"location":3.4166666666666665},"id":"26659","type":"Span"},{"attributes":{},"id":"26709","type":"BasicTickFormatter"},{"attributes":{},"id":"26588","type":"UndoTool"},{"attributes":{"source":{"id":"26642"}},"id":"26646","type":"CDSView"},{"attributes":{"data_source":{"id":"26664"},"glyph":{"id":"26665"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26666"},"selection_glyph":null,"view":{"id":"26668"}},"id":"26667","type":"GlyphRenderer"},{"attributes":{"axis":{"id":"26613"},"dimension":1,"ticker":null},"id":"26616","type":"Grid"},{"attributes":{"bottom":{"value":1},"fill_color":{"value":"#ff7f0e"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26643","type":"VBar"},{"attributes":{"axis":{"id":"26579"},"dimension":1,"ticker":null},"id":"26582","type":"Grid"},{"attributes":{"bottom":{"value":1},"fill_alpha":{"value":0.1},"fill_color":{"value":"#ff7f0e"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26644","type":"VBar"},{"attributes":{},"id":"26584","type":"PanTool"},{"attributes":{},"id":"26703","type":"UnionRenderers"},{"attributes":{},"id":"26586","type":"WheelZoomTool"},{"attributes":{},"id":"26607","type":"LinearScale"},{"attributes":{},"id":"26711","type":"Selection"},{"attributes":{"data":{"top":{"__ndarray__":"AAAAAAAA8D8OdNpApw30PxSuR+F6FPY/1AY6baDT9T8c6LSBThv4PxdLfrHkF/c/1QY6baDT9T+V/GLJL5b2P1jyiyW/WPc/43oUrkfh+T8ehetRuB75PxdLfrHkF/c/mJmZmZmZ9z8YrkfhehT2P1RVVVVVVfY/lfxiyS+W9j/gehSuR+H5P5iZmZmZmfc/kl8s+cWS9T8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26698"},"selection_policy":{"id":"26699"}},"id":"26642","type":"ColumnDataSource"},{"attributes":{},"id":"26717","type":"Selection"},{"attributes":{"data":{"top":{"__ndarray__":"ZWZmZmZm7j9OG+i0gU7XP2cDnTbQad8/WfKLJb9Y2j9Bpw102kDTP17JL5b8Yt0/PW2g0wY60T9U8oslv1jaP1ws+cWSX9w/WlVVVVVV2T9SVVVVVVXZPzTQaQOdNuA/ZgOdNtBp3z9m8oslv1jaP0h+seQXS9Y/SH6x5BdL1j84baDTBjrRPz+nDXTaQNM/SH6x5BdL1j8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26696"},"selection_policy":{"id":"26697"}},"id":"26636","type":"ColumnDataSource"},{"attributes":{},"id":"26605","type":"LinearScale"},{"attributes":{"text":"tau"},"id":"26662","type":"Title"},{"attributes":{},"id":"26718","type":"UnionRenderers"},{"attributes":{},"id":"26567","type":"DataRange1d"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"26592","type":"PolyAnnotation"},{"attributes":{"axis_label":"Rank (all chains)","formatter":{"id":"26694"},"major_label_overrides":{"0":"0","1":"1","2":"2","3":"3"},"ticker":{"id":"26576"}},"id":"26575","type":"LinearAxis"},{"attributes":{},"id":"26714","type":"UnionRenderers"},{"attributes":{},"id":"26713","type":"Selection"},{"attributes":{},"id":"26696","type":"Selection"},{"attributes":{},"id":"26571","type":"LinearScale"},{"attributes":{},"id":"26712","type":"UnionRenderers"},{"attributes":{"source":{"id":"26670"}},"id":"26674","type":"CDSView"},{"attributes":{"line_dash":[6],"location":2.480769230769231},"id":"26681","type":"Span"},{"attributes":{},"id":"26708","type":"BasicTickFormatter"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"26626","type":"PolyAnnotation"},{"attributes":{"ticks":[0,1,2,3]},"id":"26660","type":"FixedTicker"},{"attributes":{"bottom":{"value":3},"fill_alpha":{"value":0.1},"fill_color":{"value":"#d62728"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26684","type":"VBar"},{"attributes":{"line_dash":[6],"location":0.41666666666666663},"id":"26641","type":"Span"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"26591","type":"BoxAnnotation"},{"attributes":{"bottom":{"value":3},"fill_color":{"value":"#d62728"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26683","type":"VBar"},{"attributes":{"data_source":{"id":"26670"},"glyph":{"id":"26671"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26672"},"selection_glyph":null,"view":{"id":"26674"}},"id":"26673","type":"GlyphRenderer"},{"attributes":{},"id":"26576","type":"BasicTicker"},{"attributes":{"bottom":{"value":2},"fill_color":{"value":"#2ca02c"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26677","type":"VBar"},{"attributes":{"data_source":{"id":"26636"},"glyph":{"id":"26637"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26638"},"selection_glyph":null,"view":{"id":"26640"}},"id":"26639","type":"GlyphRenderer"},{"attributes":{"axis":{"id":"26575"},"ticker":null},"id":"26578","type":"Grid"},{"attributes":{"data_source":{"id":"26676"},"glyph":{"id":"26677"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26678"},"selection_glyph":null,"view":{"id":"26680"}},"id":"26679","type":"GlyphRenderer"},{"attributes":{"text":"mu"},"id":"26690","type":"Title"},{"attributes":{"bottom":{"value":2},"fill_color":{"value":"#2ca02c"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26649","type":"VBar"},{"attributes":{},"id":"26573","type":"LinearScale"},{"attributes":{"data":{"top":{"__ndarray__":"P/ADP/AD9z+SG7mRG7n2Py/0Qi/0QvU/eqEXeqEX9D9IbuRGbuT3P4If+IEf+PQ/MPRCL/RC9T+ZmZmZmZn3Pyd2Yid2YvQ/9kIv9EIv+D+4kRu5kRv7P7ATO7ETO/o/oBd6oRd6+D+mF3qhF3r4P1ZqpVZqpfk/9EIv9EIv+D9GbuRGbuT3P07sxE7sxPg/wA/8wA/8+z8=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26713"},"selection_policy":{"id":"26714"}},"id":"26670","type":"ColumnDataSource"},{"attributes":{},"id":"26716","type":"UnionRenderers"},{"attributes":{"data_source":{"id":"26682"},"glyph":{"id":"26683"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"26684"},"selection_glyph":null,"view":{"id":"26686"}},"id":"26685","type":"GlyphRenderer"},{"attributes":{},"id":"26569","type":"DataRange1d"},{"attributes":{"source":{"id":"26682"}},"id":"26686","type":"CDSView"},{"attributes":{"overlay":{"id":"26592"}},"id":"26587","type":"LassoSelectTool"},{"attributes":{"callback":null},"id":"26590","type":"HoverTool"},{"attributes":{"source":{"id":"26676"}},"id":"26680","type":"CDSView"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"26583"},{"id":"26584"},{"id":"26585"},{"id":"26586"},{"id":"26587"},{"id":"26588"},{"id":"26589"},{"id":"26590"}]},"id":"26593","type":"Toolbar"},{"attributes":{"line_dash":[6],"location":1.4807692307692308},"id":"26675","type":"Span"},{"attributes":{},"id":"26699","type":"UnionRenderers"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26637","type":"VBar"},{"attributes":{"bottom":{"value":1},"fill_color":{"value":"#ff7f0e"},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26671","type":"VBar"},{"attributes":{},"id":"26702","type":"Selection"},{"attributes":{"line_dash":[6],"location":3.480769230769231},"id":"26687","type":"Span"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26638","type":"VBar"},{"attributes":{"data":{"top":{"__ndarray__":"EPzAD/zACUAg+IEf+IELQMEP/MAPfApAdmIndmKnC0A4ciM3ciMOQIZe6IVeaA1Ah17ohV5oDUDYiZ3YiR0NQD7wAz/wAw9Ae6EXeqEXDEAbuZEbuRELQHZiJ3ZipwtAeqEXeqEXDEB0IzdyIzcLQBu5kRu5EQtAFDuxEzsxCkByIzdyIzcLQBu5kRu5EQtAxU7sxE7sCkA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26717"},"selection_policy":{"id":"26718"}},"id":"26682","type":"ColumnDataSource"},{"attributes":{},"id":"26715","type":"Selection"},{"attributes":{"below":[{"id":"26575"}],"center":[{"id":"26578"},{"id":"26582"},{"id":"26641"},{"id":"26647"},{"id":"26653"},{"id":"26659"}],"left":[{"id":"26579"}],"output_backend":"webgl","plot_height":331,"plot_width":441,"renderers":[{"id":"26639"},{"id":"26645"},{"id":"26651"},{"id":"26657"}],"title":{"id":"26662"},"toolbar":{"id":"26593"},"toolbar_location":null,"x_range":{"id":"26567"},"x_scale":{"id":"26571"},"y_range":{"id":"26569"},"y_scale":{"id":"26573"}},"id":"26566","subtype":"Figure","type":"Plot"},{"attributes":{"data":{"top":{"__ndarray__":"4Qd+4Af+BUDVSq3USq0EQBh6oRd6oQJAGHqhF3qhAkAUO7ETOzECQCu1Uiu10gRAd2IndmKnA0DFTuzETuwCQHIjN3IjNwNAJDdyIzfyA0Bu5EZu5MYCQB/4gR/4gQNAxU7sxE7sAkDTC73QCz0EQNALvdALPQRA0Au90As9BEB6oRd6oRcEQIIf+IEf+ARAhl7ohV5oBUA=","dtype":"float64","order":"little","shape":[19]},"x":{"__ndarray__":"DeU1lNdQSkDKayivobxjQCivobyGcnBAbCivobwGd0CuobyG8pp9QHkN5TWUF4JAGsprKK9hhUC8hvIayquIQF5DeQ3l9YtAAAAAAABAj0BQXkN5DUWRQKK8hvIa6pJA8hrKayiPlEBDeQ3lNTSWQJTXUF5D2ZdA5DWU11B+mUA2lNdQXiObQIbyGspryJxA2FBeQ3ltnkA=","dtype":"float64","order":"little","shape":[19]}},"selected":{"id":"26715"},"selection_policy":{"id":"26716"}},"id":"26676","type":"ColumnDataSource"},{"attributes":{"children":[{"id":"26724"},{"id":"26722"}]},"id":"26725","type":"Column"},{"attributes":{"bottom":{"value":1},"fill_alpha":{"value":0.1},"fill_color":{"value":"#ff7f0e"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26672","type":"VBar"},{"attributes":{},"id":"26701","type":"UnionRenderers"},{"attributes":{},"id":"26698","type":"Selection"},{"attributes":{"line_dash":[6],"location":0.48076923076923067},"id":"26669","type":"Span"},{"attributes":{},"id":"26694","type":"BasicTickFormatter"},{"attributes":{"bottom":{"value":2},"fill_alpha":{"value":0.1},"fill_color":{"value":"#2ca02c"},"line_alpha":{"value":0.1},"line_color":{"value":"white"},"top":{"field":"top"},"width":{"value":105.26315789473684},"x":{"field":"x"}},"id":"26678","type":"VBar"}],"root_ids":["26725"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"f0b5c336-43c9-41fd-978d-2e1def82afa0","root_ids":["26725"],"roots":{"26725":"9bf8fc82-d40e-4fab-8ace-29bb13eee093"}}];
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